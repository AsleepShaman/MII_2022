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
    public class СотрудникиController : Controller
    {
        private AudioStudioDBEntities2 db = new AudioStudioDBEntities2();

        // GET: Сотрудники
        
        public ActionResult Index()
        {
            
            return View(db.Сотрудники.ToList());
        }

        [HttpPost]
        public ActionResult Index(string Doljnost, string percent)
        {
            IEnumerable<Сотрудники> сотрудники = db.Сотрудники.AsEnumerable();
            if (Doljnost != null)
            {
                сотрудники = db.Сотрудники.Where(x => x.Должность == Doljnost);
            }
            try
            {
                if (!string.IsNullOrEmpty(percent))
                {
                    int per = Convert.ToInt32(percent);
                    using (AudioStudioDBEntities2 db = new AudioStudioDBEntities2())
                    {
                        System.Data.SqlClient.SqlParameter parameter = new System.Data.SqlClient.SqlParameter("@PerCent", per);
                        db.Database.ExecuteSqlCommand("Nalog @PerCent", parameter);
                    }
                }
            }
            catch
            {
                ViewBag.Data = "Ошибка!";
                return RedirectToAction("Error");
            }
            return View(сотрудники.ToList());
        }
        public ActionResult UserIndex(string Doljnost, string percent)
        {
            IEnumerable<Сотрудники> сотрудники = db.Сотрудники.AsEnumerable();
            if (Doljnost != null)
            {
                сотрудники = db.Сотрудники.Where(x => x.Должность == Doljnost);
            }
            try
            {
                if (!string.IsNullOrEmpty(percent))
                {
                    int per = Convert.ToInt32(percent);
                    using (AudioStudioDBEntities2 db = new AudioStudioDBEntities2())
                    {
                        System.Data.SqlClient.SqlParameter parameter = new System.Data.SqlClient.SqlParameter("@PerCent", per);
                        db.Database.ExecuteSqlCommand("Nalog @PerCent", parameter);
                    }
                }
            }
            catch
            {
                ViewBag.Data = "Ошибка!";
                return RedirectToAction("Error");
            }
            return View(сотрудники.ToList());
        }
        public ActionResult Error()
        {
            return View();
        }

        // GET: Сотрудники/Details/5
        public ActionResult Details(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Сотрудники сотрудники = db.Сотрудники.Find(id);
            if (сотрудники == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(сотрудники);
        }

        // GET: Сотрудники/Create
        public ActionResult Create()
        {
            return View();
        }

        // POST: Сотрудники/Create
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "Паспорт_сотрудника,ФИО,Должность,Дата_рождения,Телефон,Пол,Зарплата,Номер_труд__книги")] Сотрудники сотрудники)
        {
            if (ModelState.IsValid)
            {
                try
                {
                    db.Сотрудники.Add(сотрудники);
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Error!";
                    return RedirectToAction("Error");
                }
            }

            return View(сотрудники);
        }

        // GET: Сотрудники/Edit/5
        public ActionResult Edit(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Сотрудники сотрудники = db.Сотрудники.Find(id);
            if (сотрудники == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(сотрудники);
        }

        // POST: Сотрудники/Edit/5
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "Паспорт_сотрудника,ФИО,Должность,Дата_рождения,Телефон,Пол,Зарплата,Номер_труд__книги")] Сотрудники сотрудники)
        {
            if (ModelState.IsValid)
            {
               try{ 
                db.Entry(сотрудники).State = EntityState.Modified;
                db.SaveChanges();
                return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка редактирования!";
                    return RedirectToAction("Error");
                }
            }
            return View(сотрудники);
        }

        // GET: Сотрудники/Delete/5
        public ActionResult Delete(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Сотрудники сотрудники = db.Сотрудники.Find(id);
            if (сотрудники == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(сотрудники);
        }

        // POST: Сотрудники/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(string id)
        {
            try{Сотрудники сотрудники = db.Сотрудники.Find(id);
            db.Сотрудники.Remove(сотрудники);
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
