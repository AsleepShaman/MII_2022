using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using KourseWork.Models;

namespace KoursWork.Controllers
{
    public class АльбомыController : Controller
    {
        private AudioStudioDBEntities2 db = new AudioStudioDBEntities2();

        // GET: Альбомы
        public ActionResult Index()
        {
            var альбомы = db.Альбомы.Include(а => а.Исполнитель);
            return View(альбомы.ToList());
        }
        [HttpPost]
        public ActionResult Index(string janr)
        {
            IEnumerable<Альбомы> альбомы = db.Альбомы.AsEnumerable();
            if (!string.IsNullOrEmpty(janr))
            {
                альбомы = db.Альбомы.Where(x => x.Жанр == janr);
            }
            return View(альбомы.ToList());
        }
        public ActionResult UserIndex(string janr)
        {
            IEnumerable<Альбомы> альбомы = db.Альбомы.AsEnumerable();
            if (!string.IsNullOrEmpty(janr))
            {
                альбомы = db.Альбомы.Where(x => x.Жанр == janr);
            }
            return View(альбомы.ToList());
        }
        public ActionResult GuestIndex()
        {
            IEnumerable<Альбомы> альбомы = db.Альбомы.AsEnumerable();
            return View(альбомы.ToList());
        }

        public ActionResult Error()
        {
            return View();
        }

        // GET: Альбомы/Details/5
        public ActionResult Details(int? id)
        {
            if (id == null)
            {
                return RedirectToAction("Error400", "Error");
            }
            Альбомы альбомы = db.Альбомы.Find(id);
            if (альбомы == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(альбомы);
        }

        // GET: Альбомы/Create
        public ActionResult Create()
        {
            ViewBag.Паспортные_данные_исполнителя = new SelectList(db.Исполнитель, "Паспорт_Исполнителя", "ФИО");
            return View();
        }

        // POST: Альбомы/Create
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "Номер_альбома,Название_альбома,Жанр,Общая_длина,Кол_во_песен,Язык,EP,Паспортные_данные_исполнителя")] Альбомы альбомы)
        {
            if (ModelState.IsValid)
            {
                try
                {
                    db.Альбомы.Add(альбомы);
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Error!";
                    return RedirectToAction("Error");
                }
            }

            ViewBag.Паспортные_данные_исполнителя = new SelectList(db.Исполнитель, "Паспорт_Исполнителя", "ФИО", альбомы.Паспортные_данные_исполнителя);
            return View(альбомы);
        }

        // GET: Альбомы/Edit/5
        public ActionResult Edit(int? id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Альбомы альбомы = db.Альбомы.Find(id);
            if (альбомы == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            ViewBag.Паспортные_данные_исполнителя = new SelectList(db.Исполнитель, "Паспорт_Исполнителя", "ФИО", альбомы.Паспортные_данные_исполнителя);
            return View(альбомы);
        }

        // POST: Альбомы/Edit/5
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "Номер_альбома,Название_альбома,Жанр,Общая_длина,Кол_во_песен,Язык,EP,Паспортные_данные_исполнителя")] Альбомы альбомы)
        {
            if (ModelState.IsValid)
            {
                try
                {
                    db.Entry(альбомы).State = EntityState.Modified;
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка редактирования!";
                    return RedirectToAction("Error");
                }
            }
            ViewBag.Паспортные_данные_исполнителя = new SelectList(db.Исполнитель, "Паспорт_Исполнителя", "ФИО", альбомы.Паспортные_данные_исполнителя);
            return View(альбомы);
        }

        // GET: Альбомы/Delete/5
        public ActionResult Delete(int? id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Альбомы альбомы = db.Альбомы.Find(id);
            if (альбомы == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(альбомы);
        }

        // POST: Альбомы/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(int id)
        {
            try
            {
                Альбомы альбомы = db.Альбомы.Find(id);
                db.Альбомы.Remove(альбомы);
                db.SaveChanges();
                return RedirectToAction("Index");
            }
            catch
            {
                ViewBag.Data = "Ошибка удаления! Возможно, эти данные где-то есть...";
                return RedirectToAction("Error");
            }

        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
